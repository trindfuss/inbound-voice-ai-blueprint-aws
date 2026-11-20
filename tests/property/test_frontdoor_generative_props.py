"""Front-door properties for generative wording.

* For any ``say`` and ``say_lang`` the resolver returns with a card
  question (a confirmation, a disambiguation, a not-available question), the caller hears either
  ``render(say)``, when ``generative.check`` passes with that question's ``allowed`` options, or
  the catalog template; a fallback logs exactly one ``generative_fallback`` (with the check's
  reason, never the text) and an accepted text none. No digit or number word reaches TwiML from
  ``say``.
* ``generative`` takes effect only as 1.6 allows: the resolver gets
  ``generative=True``, and a valid ``say`` is spoken, exactly when the effective mode
  (``min(ceiling, configured)``, then ``generative`` only with ``generative_enabled`` on a
  non-sensitive choice slot) is ``generative``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from owlbank import callstate, segments, sensitive
from owlbank.callstate import PendingTool
from owlbank.policy import Policy, PolicyError
from owlbank.slot_policy import MODES, mode_rank
from owlbank.slots import FakeResolver, ResolverOutput, generative
from owlbank.slots.normalize import NUMBER_WORDS, lang_family
from tests.component.test_frontdoor_support import (
    DATA,
    LEES,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    says,
    sid,
)

PROPS = settings(max_examples=150, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
SEED = json.loads((DATA / "policy.json").read_text())
EN = "en-US"
NAMES = {"red": "OwlBank Red Card", "purple": "OwlBank Purple Card",
         "rewards": "OwlBank Rewards Card"}
ALL_NUMBER_WORDS = frozenset().union(*NUMBER_WORDS.values())

#: The three questions: the customer, the resolver's output (without the wording), the catalog
#: template the caller hears on a fallback, the options the wording may name, and ``asked``.
CASES: dict[str, tuple[str, ResolverOutput, str, tuple[str, ...], str]] = {
    "confirm": ("C-1001", ResolverOutput(kind="inferred", value="purple"),
                f"Did you mean the {NAMES['purple']}?", ("purple",), "confirm_option"),
    "disambiguate": ("C-1001", ResolverOutput(kind="ambiguous", candidates=("red", "purple")),
                     f"Which one did you mean: {NAMES['red']} or {NAMES['purple']}?",
                     ("red", "purple"), "disambiguate"),
    "not_available": ("C-1002", ResolverOutput(kind="inferred", value="purple"),
                      f"I don't see an {NAMES['purple']} on your profile. You have the "
                      f"{NAMES['rewards']}. Would you like to use that?",
                      ("purple", "rewards"), "not_available"),
}

_word = st.sampled_from(["the", "card", "Which", "would", "you", "like", "or", "Did", "mean", "?",
                         ",", "please", "one", "two", "uno", "una", "dos", "first", "oh", "7",
                         "²", "٣", "{", "}", "{option}", "{option:Red}", "{x:red}", "Ünö"])
_placeholder = st.sampled_from(["red", "purple", "rewards", "gold", "credit_card", "a" * 33]).map(
    lambda v: "{option:" + v + "}")
_tokens = st.lists(st.one_of(_word, _placeholder, st.text(max_size=6)), max_size=40)
#: Mostly well-formed wording (accepted, a placeholder outside ``allowed``, too long).
_clean = st.lists(st.sampled_from(["the", "{option:purple}", "{option:red}", "{option:rewards}",
                                   "card", "?", "right", "Which", "would", "you", "like"]),
                  min_size=1, max_size=60).map(" ".join)
_long = st.integers(min_value=20, max_value=40).map(lambda n: " ".join(["Which card"] * n))
says_ = st.one_of(st.none(), _clean, _clean, _long, _tokens.map(" ".join), st.text(max_size=260))
say_langs = st.sampled_from([None, EN, EN, EN, "es-US", "en", ""])


@contextmanager
def _call(n: int) -> Iterator[str]:
    call = sid(n)
    try:
        yield call
    finally:
        segments.clear_sensitive(call)
        sensitive.forget(call)


def _to_card_question(h: Harness, call: str, who: str) -> None:
    h.start(call, {"C-1001": THOMAS, "C-1002": LEES}[who])
    h.factor(call, TEST_FACTORS[who]["zip"])
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(call, TEST_FACTORS[who]["ssn4"])
    h.factor(call, OTP)
    assert h.state(call).level == 3
    # 18.13: one card on file (C-1002) needs no card question; it comes after a "no" to the
    # not-available question about a card he doesn't hold (no model call).
    args = {"from_account": "checking", "amount": "25.00",
            **({"to": "purple"} if who == "C-1002" else {})}
    callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": PendingTool(
        tool_use_id="tu_pay", name="make_payment", args=args)})
    h.handoff(call, {"reason": "confirm", "tool": "make_payment", "journal": "jr-2"})
    if who == "C-1002":
        h.xml("factor", call, Digits="2")


def _harness(out: ResolverOutput, *, ceiling: str, raw: dict[str, Any]) -> Harness:
    h = build(resolver=FakeResolver([out]), config_kw={"slot_mode_ceiling": ceiling})
    h.app.d.policy = Policy.from_dict(raw)
    return h


def _fallbacks(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == "generative_fallback"]


def _without_names(text: str) -> str:
    for name in NAMES.values():
        text = text.replace(name, " ")
    return text


def _has_number(text: str) -> bool:
    folded = generative._fold(text)  # noqa: SLF001 - the check's own folding
    if any(c.isnumeric() for c in text) or any(c.isnumeric() for c in folded):
        return True
    # 18.4: only the call's language counts (these calls are English).
    return any(t in generative.NUMBER_WORDS[lang_family(EN)] for t in re.findall(r"[^\W_]+", folded))


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _logs() -> Iterator[list[logging.LogRecord]]:
    handler = _Records()
    root = logging.getLogger()
    old = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        root.setLevel(old)


_GEN_RAW = json.loads(json.dumps(SEED))
_GEN_RAW["slot_resolution"]["generative_enabled"] = True
_GEN_RAW["slots"]["card"]["mode"] = "generative"


@PROPS
@given(case=st.sampled_from(sorted(CASES)), say=says_, say_lang=say_langs)
def test_spoken_is_the_rendered_say_or_the_template(case: str, say: str | None,
                                                          say_lang: str | None) -> None:
    who, base, template, allowed, asked = CASES[case]
    out = ResolverOutput(kind=base.kind, value=base.value, candidates=base.candidates, say=say,
                         say_lang=say_lang)
    h = _harness(out, ceiling="generative", raw=_GEN_RAW)
    with _call(300) as call:
        _to_card_question(h, call, who)
        with _logs() as records:
            root = h.xml("factor", call, SpeechResult="the violet one", Confidence="0.9")
        spoken = says(root)
        assert len(h.resolver.calls) == 1 and h.resolver.calls[0]["generative"] is True
        reason = generative.fallback_reason(say, EN, say_lang, slot="card", allowed=allowed,
                                            catalog_has=h.app.d.catalog.has)
        fbs = _fallbacks(records)
        last = [e for e in h.journal(call) if e["type"] == "prompt"][-1]
        if reason is None:
            assert say is not None
            event("accepted")
            assert spoken == [generative.render(say, "card", EN, h.app.d.catalog)]
            assert fbs == [] and last.get("generated") is True
            assert not _has_number(_without_names(spoken[0]))     # nothing numeric from say
        else:
            event(f"fallback:{reason}")
            assert spoken == [template] and "generated" not in last
            assert len(fbs) == 1
            fb = fbs[0]
            assert getattr(fb, "reason", None) == reason and getattr(fb, "asked", None) == asked
            if say and len(say.strip()) >= 12:     # long enough not to occur by chance
                for r in records:
                    assert say not in r.getMessage()
                    assert all(say not in str(v) for k, v in vars(r).items() if k != "args")


def _expected(ceiling: str, configured: str, enabled: bool, sensitive_card: bool) -> str:
    eff = configured if mode_rank(configured) <= mode_rank(ceiling) else ceiling
    if eff == "generative" and (not enabled or sensitive_card):
        return "assisted"
    return eff


@PROPS
@given(ceiling=st.sampled_from([*MODES, "generative"]), default=st.sampled_from([*MODES, "generative"]),
       card_mode=st.sampled_from([None, *MODES, "generative"]), enabled=st.booleans(),
       sensitive_card=st.booleans())
def test_generative_only_as_1_6_allows(ceiling: str, default: str, card_mode: str | None,
                                            enabled: bool, sensitive_card: bool) -> None:
    raw = json.loads(json.dumps(SEED))
    raw["slot_resolution"].update(default_mode=default, generative_enabled=enabled)
    raw["slots"]["card"].update(mode=card_mode, sensitive=sensitive_card)
    try:
        Policy.from_dict(raw)
    except PolicyError:
        assume(False)    # 1.9: generative on a sensitive slot is refused by validation
    say = "You'd like the {option:purple}, right?"
    out = ResolverOutput(kind="inferred", value="purple", say=say, say_lang=EN)
    h = _harness(out, ceiling=ceiling, raw=raw)
    configured = card_mode or default
    eff = _expected(ceiling, configured, enabled, sensitive_card)
    event(f"effective:{eff}")
    with _call(301) as call:
        _to_card_question(h, call, "C-1001")
        root = h.xml("factor", call, SpeechResult="the violet one", Confidence="0.9")
        frozen = h.session(call).flow["modes"]
        assert frozen["card"]["effective"] == eff
        assert all(d["effective"] != "generative" for n, d in frozen.items() if n != "card")
        rows = [e for e in h.journal(call) if e["type"] == "slot_resolution"]
        assert rows[-1]["slot"] == "card" and rows[-1]["effective_mode"] == eff
        if eff == "deterministic":
            assert h.resolver.calls == []
        else:
            (c,) = h.resolver.calls
            assert c["generative"] is (eff == "generative")
        generated = f"You'd like the {NAMES['purple']}, right?"
        assert (generated in says(root)) is (eff == "generative")
